import { useTour } from '@reactour/tour'
import React from 'react'
import {Navbar, Nav } from 'react-bootstrap'
import { LinkContainer } from 'react-router-bootstrap'

function Navigation() {
    const { setIsOpen } = useTour()
    return (
        <Navbar bg="light">
            <LinkContainer to="/" style={{ marginLeft: "30px" }}>
                <Navbar.Brand>ChannelExplorer</Navbar.Brand>
            </LinkContainer>
            <Navbar.Toggle aria-controls="basic-navbar-nav" />
            <Navbar.Collapse id="basic-navbar-nav">
                <Nav style={{
                    marginLeft: "auto",
                    marginRight: "30px"
                }}>
                    {/* <LinkContainer to="/featurehunt">
                        <Nav.Link>Feature Hunt</Nav.Link>
                    </LinkContainer> */}
                    <Nav.Link className='tutorial-tutorial' onClick={() => setIsOpen(true)}>Tutorial</Nav.Link>
                    <LinkContainer to="/about">
                        <Nav.Link>About</Nav.Link>
                    </LinkContainer>
                </Nav>
            </Navbar.Collapse>
        </Navbar>
    )
}

export default Navigation